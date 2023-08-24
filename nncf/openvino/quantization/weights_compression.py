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

from collections import deque
from functools import partial
from typing import Tuple, Type, Union

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.logging.logger import nncf_logger
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_operation_const_op
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_matmul_channel_axes
from nncf.openvino.statistics.statistics import OVMinMaxTensorStatistic
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.quantization.fake_quantize import calculate_scale_zero_point


def get_scale_zp_from_input_low_input_high(level_low, level_high, input_low, input_high):
    y_scale = (input_high - input_low) / (level_high - level_low)
    y_zero_point = (level_low * input_high - level_high * input_low) / (input_high - input_low)

    type_ = np.int8 if level_low < 0 else np.uint8
    level_low *= np.ones_like(y_zero_point, dtype=type_)
    level_high *= np.ones_like(y_zero_point, dtype=type_)

    max_ = np.maximum(level_low, np.round(y_zero_point).astype(type_))
    y_zero_point = np.minimum(max_, level_high)

    # y_scale = torch.squeeze(y_scale)
    # y_zero_point = torch.squeeze(y_zero_point)
    return y_scale, y_zero_point

def insert_pre_compression_operations(model: ov.Model, bits: int = 8) -> None:
    """
    Inserts in-place weights compression with FakeQuantize operation for Linear and Embedding layers.

    :param model: The original model to insert the weights compression.
    :param bits: Number of bits for quantization.
    """
    allowed_metatypes_to_const_port = {OVEmbeddingMetatype: [0], OVMatMulMetatype: [0, 1]}
    opt_125m_precisions = [8, 4, 4, 4, 4, 8, 4, 8, 4, 8, 4, 8, 4, 4, 4, 8, 4, 8, 4, 8, 4, 8, 4, 4, 4, 8, 4, 8, 4, 4, 4, 8, 8, 8, 4, 4, 4, 8, 8, 8, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 8]
    # llama2_precisions = [8, 4, 4, 8, 4, 4, 4, 4, 8, 8, 8, 4, 8, 8, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 8, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 4, 8, 4, 4, 4, 8, 8, 4, 4, 8, 4, 4, 4, 8, 4, 4, 8, 4, 4, 4, 8, 4, 8, 4, 4, 4, 8, 8, 4, 4, 8, 4, 4, 8, 8, 4, 4, 8, 4, 4, 4, 8, 4, 8, 4, 4, 4, 4, 8, 4, 4, 8, 4, 4, 4, 8, 4, 8, 4, 4, 4, 4, 8, 4, 8, 4, 4, 4, 4, 8, 4, 8, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 8, 4, 4, 8, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 8, 8, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 8, 8, 4, 8, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 8, 8, 4, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 8, 4, 8, 8, 8, 4, 4, 4, 4, 4, 8, 8, 4, 8]
    precisions = opt_125m_precisions
    # precisions = [8] * len(opt_125m_precisions)
    # precisions = [4] * len(opt_125m_precisions)
    # precisions[1] = 4
    idx = 0
    for node in model.get_ordered_ops():
        # pylint:disable=protected-access
        metatype = GraphConverter._get_node_metatype(node)
        if metatype not in allowed_metatypes_to_const_port:
            continue

        for const_port_id in allowed_metatypes_to_const_port[metatype]:
            weight_node = get_operation_const_op(node, const_port_id)
            if weight_node is None:
                continue
            # nncf_logger.info(node.get_friendly_name())
            # nncf_logger.info(weight_node.get_friendly_name())
            weight_output = weight_node.output(0)
            weight_name = weight_node.get_friendly_name()
            original_weight_dtype = weight_output.get_element_type().to_dtype()
            target_inputs = weight_output.get_target_inputs()

            if original_weight_dtype not in [np.float32, np.float16]:
                continue

            weight = get_const_value(weight_node)

            num_bits = precisions[idx]
            idx += 1
            is_power = num_bits == 4
            if is_power:
                sign = np.sign(weight)
                weight = np.sqrt(np.abs(weight)) * np.sign(weight)

            axes = _get_reduction_axes(metatype, node, const_port_id)
            min_values = np.min(weight, axis=axes, keepdims=True)
            max_values = np.max(weight, axis=axes, keepdims=True)
            stats = OVMinMaxTensorStatistic(min_values, max_values)
            quantizer_config = QuantizerConfig(
                num_bits=num_bits,
                mode=QuantizationMode.ASYMMETRIC,
                signedness_to_force=None,
                per_channel=True,
            )
            get_fq_params = partial(
                calculate_quantizer_parameters,
                quantizer_config=quantizer_config,
                quant_group=QuantizerGroup.WEIGHTS,
                narrow_range=False,
                half_range=False,
            )
            fq_params = get_fq_params(stats)

            input_low = fq_params.input_low
            input_high = fq_params.input_high
            assert np.allclose(fq_params.output_low, input_low)
            assert np.allclose(fq_params.output_high, input_high)

            levels = fq_params.levels
            # new_level_low = -levels // 2
            # new_level_high = levels - 1 + new_level_low
            new_level_low = 0 #-levels // 2
            new_level_high = levels - 1 #+ new_level_low
            print(new_level_low, new_level_high)
            scale, zero_point = calculate_scale_zero_point(
                input_low, input_high, new_level_low, new_level_high, narrow_range=False
            )
            # scale, zero_point = get_scale_zp_from_input_low_input_high(new_level_low, new_level_high, input_low, input_high)


            # int8_weight = np.round(weight / scale + zero_point).astype(np.int8)
            # int8_weight = np.clip(int8_weight, new_level_low, new_level_high).astype(np.uint8)
            # quantized_weight = opset.constant(int8_weight, dtype=np.uint8, name=weight_name)
            # convert = opset.convert(quantized_weight, original_weight_dtype)
            # zero_point = opset.constant(zero_point, dtype=np.uint8)
            # zero_point = opset.convert(zero_point, original_weight_dtype)
            # sub = opset.subtract(convert, zero_point)

            int8_weight = np.round(weight / scale + zero_point).astype(np.uint8)
            if is_power:
                quantized_weight = opset.constant(int8_weight, dtype=ov.Type.u4, name=weight_name)
            else:
                quantized_weight = opset.constant(int8_weight, dtype=np.uint8, name=weight_name)

            convert = opset.convert(quantized_weight, original_weight_dtype)
            sub = opset.subtract(convert, zero_point.astype(original_weight_dtype))
            fq_name = f"{node.get_friendly_name()}/fq_weights_{const_port_id}"
            mul_dequant_node = opset.multiply(sub, scale.astype(original_weight_dtype), name=fq_name)
            last_output = mul_dequant_node.output(0)

            if is_power:
                sign_node = opset.sign(mul_dequant_node)
                weight_shape = weight_output.shape
                pow_value = opset.constant(np.zeros(weight_shape) + 2, dtype=np.float32)
                pow_node = opset.power(mul_dequant_node, pow_value)
                mul_node = opset.multiply(pow_node, sign_node)
                last_output = mul_node.output(0)

            for target_input in target_inputs:
                target_input.replace_source_output(last_output)


    nncf_logger.info(f'number of matmuls={idx}')

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


def _insert_fake_quantize(fq_params: FakeQuantizeParameters, weight_output: ov.Output, fq_name: str, is_power: bool,
                          weight_node, const_port_id, const_value, node) -> None:
    """
    Inserts a FakeQuantize operation into the model based on the given parameters.

    :param fq_params: FakeQuantize parameters.
    :param weight_output: Output of OpenVINO node.
    :param fq_name : Name for the inserted FakeQuantize operation.
    """
    target_inputs = weight_output.get_target_inputs()

    if weight_output.get_element_type() == ov.Type(np.float16):
        input_low, input_high, output_low, output_high = OVModelTransformer.convert_params_to_fp16(fq_params)
    else:
        input_low = fq_params.input_low
        input_high = fq_params.input_high
        output_low = fq_params.output_low
        output_high = fq_params.output_high
    levels = fq_params.levels

    if is_power:
        const_value = np.sqrt(np.abs(const_value)) * np.sign(const_value)
        weight_node = _set_const_value(node, const_port_id, const_value)

        sign_node = opset.sign(weight_node)

        fq_node = opset.fake_quantize(weight_output, input_low, input_high, output_low, output_high, levels, name=fq_name)
        fq_output = fq_node.output(0)

        weight_shape = weight_output.shape
        pow_value = opset.constant(np.zeros(weight_shape) + 2, dtype=np.float32)
        pow_node = opset.power(fq_output, pow_value)

        mul_node = opset.multiply(pow_node, sign_node)
        mul_output = mul_node.output(0)

        for target_input in target_inputs:
            target_input.replace_source_output(mul_output)
    else:
        fq_node = opset.fake_quantize(weight_output, input_low, input_high, output_low, output_high, levels, name=fq_name)
        fq_output = fq_node.output(0)
        for target_input in target_inputs:
            target_input.replace_source_output(fq_output)
