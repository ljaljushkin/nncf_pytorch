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
import itertools
import json
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import openvino.runtime as ov
from numpy import linalg
from openvino.runtime import opset9 as opset

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.utils.helpers import create_table
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.node_utils import get_channel_agnostic_reduction_axes
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.rt_info import dump_parameters
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.fake_quantize import calculate_scale_zero_point
from nncf.scopes import IgnoredScope


class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    @property
    def weighted_metatypes(self) -> List[OperatorMetatype]:
        return [OVMatMulMetatype, OVEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def validate_params(mode: CompressWeightsMode, ignored_scope: Optional[IgnoredScope] = None) -> None:
        pass

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def raw_statistic_collector(inplace: bool, num_samples: int = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples, inplace)

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]

    @staticmethod
    def do_compression(
        model: ov.Model,
        nodes_to_compress: List[NNCFNode],
        mode: CompressWeightsMode,
        ratio: float = None,
        group_size: int = None,
        activations = None,
        shared_nodes_mapping = None,
        traces_per_node = None,
    ) -> ov.Model:
        all_weight_params: List[WeightNodeParams] = []
        quantized_nodes_ids = set()

        friendly_name_to_op_map = {op.get_friendly_name(): op for op in model.get_ops()}

        is_last_layer_compressed = False
        n = len(nodes_to_compress)
        save_dir = Path('saved_traces')
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(exist_ok=True)

        is_loaded_from_cache = bool(traces_per_node)
        is_hawq = traces_per_node or activations

        for i, nncf_node in enumerate(nodes_to_compress):
            node_name = nncf_node.node_name
            weight_port_ids = nncf_node.layer_attributes.get_const_port_ids()
            for weight_port_id in weight_port_ids:
                weight_op_friendly_name = nncf_node.layer_attributes.constant_attributes[weight_port_id]["name"]
                weight_node = friendly_name_to_op_map[weight_op_friendly_name]
                if weight_node is None:
                    continue
                if id(weight_node) in quantized_nodes_ids:
                    if i == n - 1:
                        is_last_layer_compressed = True
                    continue
                weight_output = weight_node.output(0)

                original_weight_dtype = weight_output.get_element_type().to_dtype()
                if original_weight_dtype not in [np.float32, np.float16, np.float64]:
                    continue
                const_shape = nncf_node.layer_attributes.constant_attributes[weight_port_id]["shape"]
                channel_axes = get_weight_channel_axes(nncf_node, weight_port_id)
                reduction_axes = get_channel_agnostic_reduction_axes(channel_axes, const_shape)
                if isinstance(reduction_axes, tuple) and len(reduction_axes) != 1:
                    nncf_logger.warning(
                        f"Weight compression expects a single reduction axes, but given {len(reduction_axes)}. "
                        f"Weight shape: {const_shape}, reduction axes: {reduction_axes}, "
                        f"node name: {node_name}. The node won't be quantized."
                    )
                    continue
                reduction_axis = reduction_axes[0] if isinstance(reduction_axes, tuple) else reduction_axes
                if is_hawq and not is_loaded_from_cache:
                    if node_name not in activations and node_name in shared_nodes_mapping:
                        print('Alles gut! shared node=', node_name)
                        original_node = shared_nodes_mapping[node_name]
                        htrace = traces_per_node[original_node]
                        traces_per_node[node_name] = htrace
                    elif node_name in activations:
                        htrace = get_hessian_trace(activations, node_name, weight_node)
                        traces_per_node[node_name] = htrace
                    elif nncf_node.metatype != OVEmbeddingMetatype and i != n - 1:
                        assert False, 'no activation found for '  + node_name

                fq_name = f"{weight_op_friendly_name}/fq_weights_{weight_port_id}"
                num_weights = np.prod(const_shape)
                weight_params = WeightNodeParams(
                    reduction_axis,
                    num_weights,
                    fq_name,
                    weight_node,
                    original_weight_dtype,
                    metatype=nncf_node.metatype,
                    htrace=traces_per_node.get(node_name)
                )
                all_weight_params.append(weight_params)
                quantized_nodes_ids.add(id(weight_node))

        if is_hawq and not is_loaded_from_cache:
            cached_traces_path = Path('traces_per_node.json')
            with open(cached_traces_path, 'w') as f:
                json.dump(traces_per_node, f)

        internal_weight_params = all_weight_params
        if mode != CompressWeightsMode.INT8:
            internal_weight_params = list(filter(lambda wp: wp.metatype != OVEmbeddingMetatype, all_weight_params))
            if not is_last_layer_compressed:
                internal_weight_params = internal_weight_params[:-1]
            primary_config = WeightCompressionConfig(mode=mode, group_size=group_size)
            if is_hawq:
                _apply_hawq(internal_weight_params, ratio, primary_config)
            else:
                _assign_mixed_precision(internal_weight_params, ratio, primary_config)
        nncf_logger.info(_get_bitwidth_distribution_str(all_weight_params, internal_weight_params))

        compression_info_per_node = {}
        for wp in all_weight_params:
            print(f"mode={wp.compression_config.mode.value} g{wp.compression_config.group_size} {wp.fq_name}")
            compression_info_per_node[wp.fq_name] = (
                wp.compression_config.mode.value,
                wp.compression_config.group_size,
            )

        for wp in track(all_weight_params, description="Applying Weight Compression"):
            weight_node = wp.weight_node
            original_weight_dtype = wp.original_weight_dtype

            weight_output = weight_node.output(0)
            weight_name = weight_node.get_friendly_name()
            target_inputs = weight_output.get_target_inputs()

            weight = get_const_value(weight_node)
            config = wp.compression_config
            if config.mode == CompressWeightsMode.NF4:
                original_shape = weight.shape
                norm_weight, scale = _get_norm_weight_and_nf4_scale(weight, wp.reduction_axis, group_size)
                compressed_const = opset.constant(norm_weight, dtype=ov.Type.nf4, name=weight_name)
                convert = opset.convert(compressed_const, original_weight_dtype)
                mul = opset.multiply(convert, scale.astype(original_weight_dtype), name=wp.fq_name)
                if config.group_size != -1:
                    mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)
                last_output = mul.output(0)
            else:
                original_shape = weight.shape
                compressed_weights, scale, zero_point = _do_integer_quantization(weight, wp.reduction_axis, config)
                compression_type = np.uint8 if config.num_bits == 8 else ov.Type.u4
                compressed_weights_node = opset.constant(compressed_weights, dtype=compression_type, name=weight_name)
                convert_weights_node = opset.convert(compressed_weights_node, original_weight_dtype)
                zero_point_node = opset.constant(zero_point, dtype=compression_type, name=f"{weight_name}/ZP")
                convert_zp_node = opset.convert(zero_point_node, original_weight_dtype)
                sub = opset.subtract(convert_weights_node, convert_zp_node)
                mul = opset.multiply(sub, scale.astype(original_weight_dtype), name=wp.fq_name)
                if config.group_size != -1:
                    mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)
                last_output = mul.output(0)

            for target_input in target_inputs:
                target_input.replace_source_output(last_output)

        dump_parameters(
            model,
            parameters={
                "mode": mode.value,
                "group_size": group_size,
                "ratio": ratio,
                "traces_per_node": traces_per_node,
                "compression_info_per_node": compression_info_per_node,
            },
            algo_name="weight_compression",
        )
        return model


TWeightType = TypeVar("TWeightType")


@dataclass
class WeightCompressionConfig:
    """
    Information on how to compress (quantize) a specific weight.

    :param mode: Defines a mode for weight compression. Defaults to INT8 mode.
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    """

    mode: Optional[CompressWeightsMode] = CompressWeightsMode.INT8
    group_size: Optional[int] = -1

    @property
    def num_bits(self):
        """
        :return: number of bits that is used for storing a single quantized value in the given mode.
        """
        return 8 if self.mode == CompressWeightsMode.INT8 else 4


@dataclass
class WeightNodeParams:
    """
    Information about weight node in the ov.Model that is useful for weight compression.

    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param num_weights: Number of elements in the weight array.
    :param fq_name: Name for the inserted weight compression operation.
    :param weight_node: The weight node itself.
    :param original_weight_dtype: Type of elements in the weight array.
    :param compression_config: Configuration of weight compression for the weight node.
    :param metatype: Metatype of the corresponding operation with weight.
    """

    reduction_axis: int
    num_weights: int
    fq_name: str
    weight_node: ov.Node
    original_weight_dtype: TWeightType
    compression_config = WeightCompressionConfig()
    metatype: OperatorMetatype = None
    htrace: Optional[np.ndarray] = None



def _do_integer_quantization(
    weight: np.ndarray, reduction_axis: int, config: WeightCompressionConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The method quantizes the given weights to integer data type in accordance with the compression config.
    The config defines a quantization mode:
        INT8 mode refers to unsigned int8 asymmetric weight compression - quantization to [0, 255] range.
        INT4_ASYM mode refers to unsigned int4 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 15] range.
        INT4_SYM mode refers to unsigned int4 symmetric weight compression with a fixed zero point equals to 8 -
            quantization to [0, 15] range.
        NF4 mode requires a dedicated procedure and it is not supported in this method.
    One of the parameter of compression config is a group size. Quantization is per-channel, if group size equals to -1,
    otherwise it's per-group, i.e. group size number of weights in the channel dimension share quantization parameters
    (scales).

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The compressed weights, scale and zero point that was used for its quantization.
    """
    mode = config.mode
    assert mode != CompressWeightsMode.NF4, "The function supports integer quantization only"
    group_size = config.group_size
    num_bits = config.num_bits

    level_low = 0
    level_high = 2**num_bits - 1

    if group_size != -1:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axis = _reshape_weights_for_grouped_quantization(weight, reduction_axis, group_size)

    if mode in [CompressWeightsMode.INT8, CompressWeightsMode.INT4_ASYM]:
        min_values = np.min(weight, axis=reduction_axis, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        max_values = np.max(weight, axis=reduction_axis, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        scale, zero_point = calculate_scale_zero_point(
            min_values, max_values, level_low, level_high, narrow_range=False
        )
    else:
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
        level_low_sym = -(2 ** (num_bits - 1))
        level_high_sym = 2 ** (num_bits - 1) - 1
        scale = scale / level_high_sym
        zero_point = np.array([-level_low_sym])

    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    compressed_weights = np.round(weight / scale + zero_point)
    compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)
    return compressed_weights, scale, zero_point

def _get_l2norm_of_quant_noise(weight: np.ndarray, reduction_axis: int, config: WeightCompressionConfig) -> float:
    """
    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    """
    orig_shape = weight.shape
    compressed_weights, scale, zero_point = _do_integer_quantization(weight, reduction_axis, config)

    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (compressed_weights - zero_point) * scale

    decompressed_weight = decompressed_weight.reshape(orig_shape)
    return linalg.norm(decompressed_weight - weight, ord='fro')

def _get_integer_quantization_error(weight: np.ndarray, reduction_axis: int, config: WeightCompressionConfig) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and fake quantized
    (compressed and decompressed) to integer ones.

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The quantity characterizing the error of integer quantization.
    """
    orig_shape = weight.shape
    compressed_weights, scale, zero_point = _do_integer_quantization(weight, reduction_axis, config)

    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (compressed_weights - zero_point) * scale

    decompressed_weight = decompressed_weight.reshape(orig_shape)
    diff = (decompressed_weight - weight) ** 2
    layer_err = np.mean(diff, axis=reduction_axis)
    val = np.max(layer_err)
    return val

def _transpose_for_matmul(t: np.ndarray):
    a=list(range(len(t.shape)))
    # transpose two right-most axes
    a[-1], a[-2] = a[-2], a[-1]
    return np.transpose(t, axes=a)

def _do_matmul(x, w, transpose_x: bool = False, transpose_w: bool = False):
    # if transpose_x:
    #     x = _transpose_for_matmul(x)
    # if transpose_w:
    #     w = _transpose_for_matmul(w)
    return np.matmul(x,w)


def _reshape_weights_for_grouped_quantization(
    weight: np.ndarray, reduction_axis: int, group_size: int
) -> Tuple[np.ndarray, int]:
    """
    Reshapes weights for group-wise quantization and return a new reduction axis for collecting statistics per group
    dimension. Having weights with shapes [c_out, c_in] and group size = 128, shape of reshaped weights is
    [c_out, c_in // 128, 128].

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weights and new reduction axis.
    """
    assert group_size != -1
    assert isinstance(reduction_axis, int)
    channel_size = weight.shape[reduction_axis]
    if channel_size % group_size != 0:
        raise RuntimeError(f"Channel size {channel_size} should be divisible by size of group {group_size}")

    num_groups_per_channel = channel_size // group_size
    shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axis : reduction_axis + 1] = (num_groups_per_channel, group_size)
    reshaped_weight = weight.reshape(shape)
    reduction_axis += 1
    return reshaped_weight, reduction_axis


def _get_norm_weight_and_nf4_scale(
    weight: np.ndarray, reduction_axis: int, group_size: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates scale for nf4 quantization and normalizes weights by the scale.
    Weights are reshaped in case of positive value of group size.

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    :return: Normalized weights and nf4 scale.
    """
    if group_size != -1:
        # weights are reshaped: [a1, r, a2] -> [a1, r//gs, gs, a2]
        weight, reduction_axis = _reshape_weights_for_grouped_quantization(weight, reduction_axis, group_size)
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
    else:
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, 1, a2]
    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    norm_weight = weight / scale
    return norm_weight, scale


def _proportion_str(num_weights_list: List[int], total_num_weights: int, total_num_params: int) -> str:
    percentage = sum(num_weights_list) / max(total_num_weights, 1) * 100
    return f"{percentage:.0f}% ({len(num_weights_list)} / {total_num_params})"


def _get_bitwidth_distribution_str(all_params: List[WeightNodeParams], internal_params: List[WeightNodeParams]) -> str:
    """
    Generates a table that shows the ratio of weights quantized to different number of bits.

    :param all_params: List of information about each weight node.
    :param internal_params: List of information about weight nodes that are considered for mixed precision.
    :return: A string containing the table.
    """
    not_internal_params = [wp for wp in all_params if wp not in internal_params]
    num_bits_vs_num_weights_map = {}
    for data in internal_params:
        num_bits = data.compression_config.num_bits
        n_internal, n_internal = num_bits_vs_num_weights_map.get(num_bits, ([], []))
        n_internal.append(data.num_weights)
        num_bits_vs_num_weights_map[num_bits] = (n_internal, n_internal)
    for data in not_internal_params:
        num_bits = data.compression_config.num_bits
        n_total, n_internal = num_bits_vs_num_weights_map.get(num_bits, ([], []))
        n_total.append(data.num_weights)
        num_bits_vs_num_weights_map[num_bits] = (n_total, n_internal)
    num_internal_weights = sum(ws.num_weights for ws in internal_params)
    num_internal_params = len(internal_params)
    total_num_weights = num_internal_weights + sum(ws.num_weights for ws in not_internal_params)
    num_params = len(all_params)
    num_bits_vs_num_weights_map = OrderedDict(sorted(num_bits_vs_num_weights_map.items(), reverse=True))
    # Table creation
    header = ["Num bits (N)", "% all parameters (layers)", "% internal parameters (layers)"]
    rows = []
    for bitwidth, (n_total, n_internal) in num_bits_vs_num_weights_map.items():
        rows.append(
            [
                bitwidth,
                _proportion_str(n_total, total_num_weights, num_params),
                _proportion_str(n_internal, num_internal_weights, num_internal_params),
            ]
        )

    table = create_table(header, rows)
    pretty_string = f"Statistics of the bitwidth distribution:\n{table}"
    return pretty_string

def get_hessian_trace(activations, node_name, weight_node):
    # TODO: handle last layer?? should be skipped early
    # TODO: any way to accelerate?? need diagonal elements only!!! multiply 1st row with 1st column, 2nd row with 2nd column, etc ...
    list_acts = activations[node_name]
    weight = get_const_value(weight_node)
    orig_shape = weight.shape
    print('weight shape: ', orig_shape, ' activation shape: ', list_acts[0].shape, ' name: ', node_name)

    columns = orig_shape[1]
    # H = np.zeros((columns, columns))
    # print('hessian shape: ', H.shape)
    htrace = 0
    nsamples = 0
    for inp in list_acts:
        # if len(inp.shape) == 2:
        #     inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        if False:
            inp = np.transpose(inp) # [S, H] -> [H, S]
            # TODO: is it properly normalized??? Hessian for FC2, FC1 in opt-125m has small values, because of dimensions??
            H *= nsamples / (nsamples + tmp)
            nsamples += tmp
            inp = np.sqrt(2 / nsamples) * inp
            # TODO: avoid double transpose: np.matmul(np.transpose(inp), inp)
            H += np.matmul(inp, np.transpose(inp)) # [H, S] * [S, H] -> [H, H]
        else:
            htrace *= nsamples / (nsamples + tmp)
            nsamples += tmp
            inp = np.sqrt(2 / nsamples) * inp
            # NOTE: need diagonal elements of Hessian only for trace, no need to do full matrix multiplication!
            # TODO: check for batch_size != 1 with reshape !!! will it really sum diagonal elements?
            htrace += np.sum(np.multiply(inp, inp))
    # TODO: consider the full estimation, not just trace: deltaW * Hessian * delta.t()
    # htrace = np.trace(H)
    return htrace

def _assign_mixed_precision(
    internal_weight_params: List[WeightNodeParams], ratio: float, primary_config: WeightCompressionConfig
) -> None:
    """
    Assigns mixed quantization scheme (e.g. uniform int8 or non-uniform nf4) for weights based on some criteria.
    :param internal_weight_params: List of information about internal weight nodes. Only internal nodes are considered
        for mixed precision. The quantization scheme is added to this info.
    :param ratio: The ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8).
    :param primary_config: Information on how to compress (quantize) weights to primary precision.
    :return: None.
    """
    if ratio == 1:
        for weight_param in internal_weight_params:
            weight_param.compression_config = primary_config
        return
    errors = []
    num_internal_weights = 0
    # NOTE: first and last layers are always in 8 bit: no need to calculate error for them
    for weight_param in track(internal_weight_params, description="Searching for Mixed-Precision Configuration"):
        weight = get_const_value(weight_param.weight_node)
        backup_config = weight_param.compression_config
        reduction_axis = weight_param.reduction_axis
        backup_error = _get_integer_quantization_error(weight, reduction_axis, backup_config)
        eps = np.finfo(weight.dtype).eps
        error = 1 / (backup_error + eps)
        errors.append(error)
        num_internal_weights += weight_param.num_weights

    fig, ax = plt.subplots()
    ax.set_title('int8 error per layer')
    ax.set_xlabel('Layers')
    ax.plot(errors)
    plt.savefig("int8_error.png")

    indexes_of_layers_in_ascending_order_of_errors = [
        i[0] for i in sorted(enumerate(errors), reverse=False, key=lambda x: x[1])
    ]
    num_weights_in_4bit = 0
    for index in indexes_of_layers_in_ascending_order_of_errors:
        weight_param = internal_weight_params[index]
        current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_internal_weights
        if current_ratio >= ratio:
            break
        weight_param.compression_config = primary_config
        num_weights_in_4bit += weight_param.num_weights

# def get_all_non_decreasing_bitwidth_sequences(number_of_layers) -> List[List[int]]:
#     start_time = time.time()
#     sequences = []
#     bitwidths_ = [4, 8]
#     seq_len = number_of_layers
#     if seq_len == 0:
#         return sequences
#     bitwidths = sorted(bitwidths_)
#     m = len(bitwidths)
#     L = seq_len
#     for j in range(1, m + 1):
#         for combo_bitwidths in itertools.combinations(bitwidths, j):
#             for combo_partitions in itertools.combinations(list(range(1, L)), j - 1):
#                 bit_config = []
#                 prev_p = 0
#                 for p, b in zip(combo_partitions + (L,), combo_bitwidths):
#                     bit_config += [b] * (p - prev_p)
#                     prev_p = p
#                 sequences.append(bit_config)
#     print('Collecting {} bitwidth sequences (out of {} layers) takes {:3.2f} minutes'.format(len(sequences),number_of_layers, (time.time() - start_time) / 60))
#     return sequences

# def calc_hawq_metric_per_bitwidth_sequence(_bitwidth_sequences, int4_errors, int8_errors, traces):
#     start_time = time.time()
#     # TODO: check and draw!!
#     metric_per_bitwidth_sequence = []
#     error_map = {4: int4_errors, 8: int8_errors}
#     for bitwidth_sequence in _bitwidth_sequences:
#         hawq_metric = np.zeros(1)
#         # can be vectorized by getting list indexes
#         for index, bitwidth in enumerate(bitwidth_sequence):
#             quant_noise = error_map[bitwidth][index]
#             trace = traces[index]
#             hawq_metric += quant_noise * trace
#         metric_per_bitwidth_sequence.append(hawq_metric)
#     print('Calculate hawq metrics takes {:3.1f} minutes'.format((time.time() - start_time) / 60))
#     return metric_per_bitwidth_sequence


def _apply_hawq(
    internal_weight_params: List[WeightNodeParams], ratio: float, primary_config: WeightCompressionConfig
) -> None:
    """
    TODO:
    :param internal_weight_params: List of information about internal weight nodes. Only internal nodes are considered
        for mixed precision. The quantization scheme is added to this info.
    :param ratio: The ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8).
    :param primary_config: Information on how to compress (quantize) weights to primary precision.
    :return: None.
    """
    start_time = time.time()
    if ratio == 1:
        for weight_param in internal_weight_params:
            weight_param.compression_config = primary_config
        return

    # int4_errors = []
    l2norm_noises = []
    num_internal_weights = 0
    traces = []
    perturbations = []
    # start_time = time.time()
    # TODO: is cache really needed?? for llama-7b definitely!
    for weight_param in track(internal_weight_params, description="Collecting quantization noise"):
        weight = get_const_value(weight_param.weight_node)
        backup_config = weight_param.compression_config
        reduction_axis = weight_param.reduction_axis
        l2norm_noise = _get_l2norm_of_quant_noise(weight, reduction_axis, backup_config)
        eps = np.finfo(weight.dtype).eps
        # NOTE: calc real perturbation - no need to normalize?
        # int8_error = 1 / (int8_error + eps)
        l2norm_noises.append(l2norm_noise)
        # int4_error = _get_integer_quantization_error(weight, reduction_axis, primary_config)
        # eps = np.finfo(weight.dtype).eps
        # error = 1 / (int4_error + eps)
        # int4_errors.append(error)

        num_internal_weights += weight_param.num_weights
        trace = weight_param.htrace
        traces.append(trace)
        perturbations.append(trace * l2norm_noise)
    print('Collecting perturbation takes {:3.1f} minutes'.format((time.time() - start_time) / 60))

    fig, ax = plt.subplots()
    ax.plot(traces, label='traces')
    ax.plot(perturbations, label='perturbations')
    ax.set_title('Trace/Perturbation per layer')
    ax.set_xlabel('Layers')
    ax.set_ylabel('Metric value')
    ax.legend(loc='upper right')
    plt.savefig("perturbations.png")

    fig, ax = plt.subplots()
    ax.set_title('L2Norm of quantization noise per layer')
    ax.set_xlabel('Layers')
    ax.plot(l2norm_noises, label='l2norm_8bit_noise')
    plt.savefig("l2norm_noise.png")

    indexes_of_layers_in_ascending_order_of_perturbations = [
        i[0] for i in sorted(enumerate(perturbations), reverse=False, key=lambda x: x[1])
    ]

    # _bitwidth_sequences = get_all_non_decreasing_bitwidth_sequences(len(internal_weight_params))
    # metric_per_bitwidth_sequence = calc_hawq_metric_per_bitwidth_sequence(_bitwidth_sequences, int4_errors, int8_errors, traces, indexes_of_layers_in_ascending_order_of_traces)
    # assert False, 'OK!'
    # indexes_of_layers_in_ascending_order_of_errors = [
    #     i[0] for i in sorted(enumerate(errors), reverse=False, key=lambda x: x[1])
    # ]

    num_weights_in_4bit = 0
    for index in indexes_of_layers_in_ascending_order_of_perturbations:
        weight_param = internal_weight_params[index]
        current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_internal_weights
        if current_ratio >= ratio:
            break
        weight_param.compression_config = primary_config
        num_weights_in_4bit += weight_param.num_weights
